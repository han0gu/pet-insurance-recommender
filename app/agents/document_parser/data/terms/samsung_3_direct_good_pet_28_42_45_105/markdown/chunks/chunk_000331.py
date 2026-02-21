from langchain_core.documents import Document

chunk = Document(
    page_content=('- 금 지급사유조사와 관련하여 의료기관, 동물병원, 국민건강보험공단, 경찰서 등 관공\n'
 '- 서에 대한 회사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정당한 사유없이\n'
 '- 이에 동의하지 않을 경우 사실확인이 끝날 때까지 회사는 보험금 지급 지연에 따른\n'
 '- 이자를 지급하지 않습니다.\n'
 '- ⑥ 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 설명합\n'
 '- 니다.\n'
 '# 제 10조 (보험금의 분담)① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합'),
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
