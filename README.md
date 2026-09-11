# left-rl-with-mujoco
xml 수정 사항
1. 몸체 부분 고정
2. 큐브 낙하 > free joint를 slide joint 2개로 교체
3. robot.xml에 손가락 자기 자신의 링크끼리 충돌 제외(self-collision exclude)가 하나도 없어서 액추에이터가 동작해도 튕겨나가는 문제 > 손가락별 인접링크 exclude 16쌍을 추가해서 수정 (손가락-베이스 간 충돌 제외는 존재)
4. robot.xml의 collision geom 17개를 전부 mesh → box primitive로 교체 (STL bounding box 기준, visual mesh는 그대로 유지 — 눈에 보이는 모양은 안 바뀌고 충돌 계산만 단순화됨). 굽힘/그립 동작은 여전히 정상 (렌더링으로 확인)
