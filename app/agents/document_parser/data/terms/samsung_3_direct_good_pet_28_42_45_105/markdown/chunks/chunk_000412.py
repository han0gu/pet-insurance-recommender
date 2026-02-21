from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 이 특별약관에서「이물제거(구토유발약물)」란 반려견의 위장 등 내부의 이물질을 제\n'
 '- 거하기 위하여 수술 또는 내시경을 동반하지 않고 구토유발을 목적으로 한 약물을 이\n'
 '- 용한 의료행위를 말합니다.\n'
 "<유의사항># [수술]동물병원의 수의사 자격을 가진 자(이하 '수의사'라 합니다)에 의하여 치료가 필요하다고 인정된 상\n"
 '해 또는 질병 치료를 위하여 수의사법 제 17조(개설)에서 규정한 국내의 동물병원에서 수의사의'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
