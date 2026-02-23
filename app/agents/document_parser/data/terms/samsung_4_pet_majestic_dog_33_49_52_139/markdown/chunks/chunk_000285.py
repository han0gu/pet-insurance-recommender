from langchain_core.documents import Document

chunk = Document(
    page_content=('손해를 배상할 책임을 집니다.\n'
 '③ 회사가 보험금 지급여부 및 지급금액에 관하여 현저하게 공정을 잃은 합의로 보험수\n'
 '익자에게 손해를 가한 경우에도 회사는 제2항에 따라 손해를 배상할 책임을 집니다.- \n'
 '<용어풀이># [현저하게 공정을 잃은 합의]회사가 보험수익자의 경제적․신체적․정신적인 어려움, 경솔함, 경험 부족 등을 이용하여 '
 '동일․유사\n'
 '사례에 비추어 보험수익자에게 매우 불합리하게 합의를 하는 것을 의미합니다.# 제44조 (개인정보보호)① 회사는 이 특별약관과 관련된 '
 '개인정보를 이 특별약관의 체결, 유지, 보험금 지급 등'),
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
