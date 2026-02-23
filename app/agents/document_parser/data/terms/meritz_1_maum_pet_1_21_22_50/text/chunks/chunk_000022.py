from langchain_core.documents import Document

chunk = Document(
    page_content=('아(심장사상충) 감염, 인플루엔자 감염, 고양이범백혈구감소증, 고양이칼리시바이러\n'
 '스감염증, 고양이바이러스성비기관지염, 고양이백혈병바이러스감염증, 고양이헤르페\n'
 '스바이러스감염증, 고양이클라미디아감염증\n'
 '3. 보험증권에 기재된 반려동물이 개(犬)인 경우, 반려동물의 슬관절탈구, 고관절탈구,\n'
 '슬관절형성부전, 고관절형성부전(대퇴 골두 허혈성 괴사 포함) 또는 기타 이들과 유\n'
 '사한 질병 또는 상해\n'
 '4. 보험증권에 기재된 반려동물이 고양이(猫)인 경우, 반려동물의 비뇨기계질환, 전염성\n'
 '복막염 또는 기타 이들과 유사한 질병 또는 상해'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
