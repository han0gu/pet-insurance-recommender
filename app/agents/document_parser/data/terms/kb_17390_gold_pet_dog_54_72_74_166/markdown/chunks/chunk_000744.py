from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제6조(보험금의 분담)\uf000 회사는 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약\n'
 '을 포함합니다)이 있을 경우 각 계약에 대하여 다른 계약이 없는 것으로 하여 각\n'
 '각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를\n'
 '보상합니다.다른 계약이 없을 때# \uf000| 피보험자가 부담한 위탁비용 피보험자가 다른 계약에 | × 대하여 | 이 계약의 지급보험금 '
 '다른 계약이 없는 것으로 하여 |\n'
 '| --- | --- | --- |'),
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
