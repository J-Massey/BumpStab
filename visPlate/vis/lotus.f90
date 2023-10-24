program flat_plate
  !
    use bodyMod,    only: body
    use fluidMod,   only: fluid
    use mympiMod,   only: init_mympi,mympi_end,mympi_rank
    use gridMod,    only: xg,composite
    use imageMod,   only: display
    use geom_shape
    implicit none
  !
  ! -- Physical parameters
    real,parameter     :: Re = 1.2e4
  !
    real,parameter     :: c = 384.0, nu = c/Re
    real, parameter    :: finish = 220.0
    integer            :: b(3) = [2,4,2]
  !
  ! -- Hyperparameters
    real, parameter    :: alpha=pi*8/180, thicc=c/45.71, d=58*c/960, pro=3*c/320, l_c=7.*c/160. ! - bump diameter 7mm, protrude 1.5mm
  !
  ! -- Dimensions
    integer            :: n(3), ndims = 3
  !
  ! -- Setup solver
    logical            :: there = .false., root, p(3) = (/.FALSE.,.FALSE.,.FALSE./)
    real               :: m(3), z
    type(fluid)        :: flow
    type(body)         :: geom
    
  !
  ! -- Outputs
    real            :: dt, t, pforce(3), vforce(3), cf_s(3)
  !
  !
  ! -- Initialize
    call init_mympi(ndims,set_blocks=b(1:ndims),set_periodic=p(1:ndims))
    root = mympi_rank()==0
    if(root) print *,'Setting up the grid, body and fluid'
    if(root) print *,'-----------------------------------'
  !
  ! -- Setup the grid
    if(ndims==3) then
      z = 1.
    else
      z = 0.
    end if
    m = [2.0,2.0, z]
    n = composite(c*m,prnt=root)
    call xg(1)%stretch(n(1), -5.*c, -1.*c, 2.*c, 9.*c,  h_min=4., h_max=10., prnt=root)
    call xg(2)%stretch(n(2), -3.*c, -0.8*c, 1.5*c, 3.*c, h_min=1.5, prnt=root)
    if(ndims==3) xg(3)%h = 3.
  !
  ! -- Call the geometry and kinematics
    geom = bumps_top(c,thicc,d,pro,l_c).or.bumps_bottom(c,thicc,d,pro,l_c).or.wavy_wall(c, thicc)
  !
  ! -- Initialise fluid
    call flow%init(n/b,geom,V=(/cos(alpha),sin(alpha),0./),nu=nu,exit=.true.)
    ! flow%time = 0
  !
    if(root) print *,'Starting time update loop'
    if(root) print *,'-----------------------------------'
    if(root) print *,' -t- , -dt- '
  !
    time_loop: do while (flow%time/c<finish.and..not.there)
      dt = flow%dt                        ! time step
      call geom%update(flow%time+flow%dt) ! update geom
      call flow%update(geom)              ! update N-S
      t = flow%time
      
      
      pforce = -2.*geom%pforce(flow%pressure)/(thicc*sin(alpha)*n(3)*xg(3)%h)
      vforce = -2.*nu*geom%vforce_s(flow%velocity)/(thicc*sin(alpha)*n(3)*xg(3)%h)

      if((mod(t,5.*c)<dt).and.(root)) print "('Time:',f15.3,'. Time remaining:',f15.3,3f12.6)",t/c,finish-t/c
      
      ! if((mod(t,0.05*c)<dt).and.(t/c>finish-40)) call flow%write(write_vtr=.false.)
      if((mod(t,60.*c)<dt)) call flow%write()

      if(root) then
        write(9,'(f10.4,f8.4,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8,4e16.8)') &
         t,dt,pforce,vforce
        flush(9)
      end if

      inquire(file='../.kill', exist=there)
      if (there) exit time_loop
      exit

    end do time_loop

    if(root) print *,'Loop complete: writing restart files and exiting'
    if(root) print *,'-----------------------------------'

    call flow%write(geom, write_vtr=.false.)
    call mympi_end

  contains
  !
  type(set) function wavy_wall(length, thickness) result(geom)
  real,intent(in) :: length, thickness
  real,parameter  :: s2 = 1.
  geom = plane([0.,1.,0.],[0.,0.5*thickness,0.])&! upper surface
  .and.plane([-0.,-1.,0.],[0.,-0.5*thickness,0.]) &  ! lower surface
  .and.plane([1.,-0.,0.],[0.5*length,0.,0.]) & ! end cap
  .and.plane([-1.,0.,0.],[-0.5*length,0.,0.]) ! front cap
  end function

  type(set) function bumps_top(l, t, d, pro, l_c) result(geom)
    real,intent(in) :: l, t, d, pro, l_c
    real,parameter  :: s2 = 1.
    !
    ! -- Define a circle at the correct position relative to the foil 
    ! -- Slice it so that the top and bottom bumps won't overlap
    ! -- Then slice it so the packing doesn't cause adjascent circles to overlap
    ! -- l_c is the chord length of the sphere
    !
    geom = sphere(radius=d/2, center=[l_c/2,(-d/2+pro+thicc/2),l_c/2]) &
    .and.plane([-1.,0.,0.],[0.,0.,0.]).and.plane([+1.,0.,0.],[l_c,0.,0.])&      ! - slice x
    .and.plane([0.,-1.,0.],[0.,0.,0.])&                                         ! - slice y
    .and.plane([0.,0.,-1.],[0.,0.,0.]).and.plane([0.,0.,+1.],[0.,0.,l_c])&      ! - slice z
    .map.init_repeat(axis=1,mod=1.01*l_c,low=-0.5*c+0.4*l_c,high=0.5*c-0.4*l_c)&
    .map.init_repeat(axis=3,mod=1.06*l_c)
  end function

  type(set) function bumps_bottom(l, t, d, pro, l_c) result(geom)
    real,intent(in) :: l, t, d, pro, l_c
    real,parameter  :: s2 = 1.
    !
    ! -- Define a circle at the correct position relative to the foil 
    ! -- Slice it so that the top and bottom bumps won't overlap
    ! -- Then slice it so the packing doesn't cause adjascent circles to overlap
    ! -- l_c is the chord length of the sphere
    !
    geom = sphere(radius=d/2, center=[l_c/2,-(-d/2+pro+thicc/2),l_c/2]) &
    .and.plane([-1.,0.,0.],[0.,0.,0.]).and.plane([+1.,0.,0.],[l_c,0.,0.])&      ! - slice x
    .and.plane([0.,+1.,0.],[0.,0.,0.])&                                         ! - slice y
    .and.plane([0.,0.,-1.],[0.,0.,0.]).and.plane([0.,0.,+1.],[0.,0.,l_c])&      ! - slice z
    .map.init_repeat(axis=1,mod=1.01*l_c,low=-0.5*c+0.4*l_c,high=0.5*c-0.4*l_c)&! no half bumps
    .map.init_repeat(axis=3,mod=1.06*l_c)
  end function
  
  end program flat_plate
  
