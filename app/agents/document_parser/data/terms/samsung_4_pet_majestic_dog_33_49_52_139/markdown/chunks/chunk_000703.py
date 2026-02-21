from langchain_core.documents import Document

chunk = Document(
    page_content=('- 출퇴근용도 등으로 주로 사용하는 경우에 한하며 일회적인 사용은 제외), 관리하는 경\n'
 '- 우에 한하여 부가하여 이루어 집니다.\n'
 '- ④ 보험계약이 해지, 기타사유에 의하여 효력이 없게 된 경우에는 이 특별약관도 더 이상\n'
 '- 효력이 없습니다.\n'
 '# 제2조 (보험금을 지급하지 않는 사유)- ① 회사는 보험계약의 내용에도 불구하고 피보험자가 보험기간 중에 이륜자동차를 운전\n'
 '- (탑승을 포함합니다. 이하 같습니다)하는 중에 발생한 급격하고도 우연한 외래의 상해'),
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
