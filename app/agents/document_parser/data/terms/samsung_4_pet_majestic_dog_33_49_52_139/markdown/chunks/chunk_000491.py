from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이자를 지급하지 않습니다.\n'
 '- ⑥ 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 설명합\n'
 '- 니다.\n'
 '# 제 10조 (보험금의 분담)① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합\n'
 '니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각각 산출한 보상\n'
 '책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.| 피보험자가 부담한 의료비 × | <지급보험금 계산방법> '
 '다른 계약이 없을 때 이 계약의 지급보험금 |'),
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
