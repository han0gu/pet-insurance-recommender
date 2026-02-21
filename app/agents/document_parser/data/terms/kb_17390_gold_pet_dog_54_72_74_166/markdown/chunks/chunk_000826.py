from langchain_core.documents import Document

chunk = Document(
    page_content=('- 단, 유사계약 청약일 이후 제1항에서 정한 질병과 관련한 새로운 위험(재진단·치\n'
 '- 료 등은 해당하지 않습니다)이 발생하거나, 새로운 질병에 대한 보장이 추가(입원\n'
 '- 비, 수술비, 진단비 등 보장 범위의 변경 또는 확대는 해당하지 않습니다)된 경우\n'
 '- 이를 적용하지 않을 수 있습니다.\n'
 '- \uf000 제1항에도 불구하고 다음 사항 중 어느 한가지의 경우에 해당되는 사유로 보험계\n'
 '- 약에서 정한 보험금의 지급사유가 발생한 경우에는 보험금을 지급합니다.\n'
 '- 1. 제1항에서 지정한 특정질병의 합병증으로 인해 발생한 특정질병이외의 질병'),
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
