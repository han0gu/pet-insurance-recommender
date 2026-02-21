from langchain_core.documents import Document

chunk = Document(
    page_content=('- 임 및 인공수정 관련 합병증으로 인한 경우에는 보험\n'
 '- 금을 지급합니다.\n'
 '# 【습관성 유산, 불임 및 인공수정】한국표준질병·사인분류상의 N96~N98에 해당하는 질병을\n'
 '말합니다.# ⑤ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동\uf000 회사는 다른 약정이 없으면 피보험자가 직업, 직무 또는\n'
 '동호회 활동목적으로 아래에 열거된 행위로 인하여 제3조55(보험금의 지급사유)의 상해 관련 보험금 지급사유가 발생\n'
 '한 때에는 해당 보험금을 지급하지 않습니다.- ① 전문등반(전문적인 등산용구를 사용하여 암벽 또는 빙'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
