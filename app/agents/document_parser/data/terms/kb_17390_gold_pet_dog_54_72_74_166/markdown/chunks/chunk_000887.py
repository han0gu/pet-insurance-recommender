from langchain_core.documents import Document

chunk = Document(
    page_content=('# 남긴 때- \n'
 '나.장해판정기준\n'
 '1) ‘체간골’이라 함은 어깨뼈(견갑골), 골반뼈(장골, 제2천추 이하의 천\n'
 '골, 미골, 좌골 포함), 빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골)를\n'
 '말하며 이를 모두 동일한 부위로 본다.\n'
 '2) ‘골반뼈의 뚜렷한 기형’이라 함은 아래의 경우 중 하나에 해당하는 때\n'
 '를 말한다.- \n'
 '가) 천장관절 또는 치골문합부가 분리된 상태로 치유되었거나 좌골이\n'
 '2.5cm이상 분리된 부정유합 상태\n'
 '나) 육안으로 변형(결손을 포함)을 명백하게 알 수 있을 정도로 방사선'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
