# Dot-source this file in the ROS request shell. Loading sends no ROS requests.
$execution = 'left-test-' + [guid]::NewGuid().ToString('N')
$quarter = 1
$dry = 'true'

function Send-LeftGoal {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('grasp', 'rotate', 'release')][string]$Primitive,
        [Parameter(Mandatory = $true)][int]$Turns,
        [Parameter(Mandatory = $true)][int]$Step
    )
    if ($quarter -notin @(1, -1, 2)) { throw 'quarter must be 1, -1 or 2' }
    if ($dry -notin @('true', 'false')) { throw 'dry must be true or false' }
    if ($Step -lt 1) { throw 'Step must be positive' }
    if (($Primitive -eq 'rotate' -and $Turns -notin @(1, -1, 2)) -or
        ($Primitive -ne 'rotate' -and $Turns -ne 0)) { throw 'Invalid turn count' }
    # A direct rotate -1/2 call must describe the same move as its turn count.
    $goalQuarter = if ($Primitive -eq 'rotate') { $Turns } else { $quarter }
    $move = switch ($goalQuarter) {
        1  { 'U' }
        -1 { "U'" }
        2  { 'U2' }
    }
    $goal = "{scan_id: bench, execution_id: $execution, move_index: 1, step_index: $Step, cube_move: $move, target_face: U, face_direction_order: URFDLB, target_face_direction_index: 0, primitive_id: ${Primitive}_upper_layer, clockwise_quarter_turns: $Turns, dry_run: $dry}"
    Write-Host "execution_id=$execution dry_run=$dry primitive=$Primitive turns=$Turns"
    & ros2 action send_goal /custom_hand/execute_primitive system_interfaces/action/ExecuteHandPrimitive $goal --feedback
}
Write-Host 'Commands loaded. No request sent. Default: dry_run=true, quarter=1.'
Write-Host 'For hardware testing, explicitly set $dry = ''false'' before sending a goal.'
