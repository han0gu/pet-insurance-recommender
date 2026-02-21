from langchain_core.documents import Document

chunk = Document(
    page_content=('- 험료 납입면제) 및 제6조(보험료 납입면제에 관한 세부규정)를 적용하지 않습니다.\n'
 '- ③ 제1항에도 불구하고, 특별약관 일반사항 제35조(해약환급금)를 적용하지 않으며 보통\n'
 '- 약관 제36조(해약환급금)을 적용합니다.\n'
 '# 제2관 개별사항# 제1조 (보험금의 지급사유)회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합\n'
 '니다) 중에 상해로 장해분류표([별표2]장해분류표 참조. 이하 같습니다)에서 정한\n'
 '3~100% 장해지급률에 해당하는 장해상태가 되었을 때에는 장해분류표에서 정한 지급률'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
