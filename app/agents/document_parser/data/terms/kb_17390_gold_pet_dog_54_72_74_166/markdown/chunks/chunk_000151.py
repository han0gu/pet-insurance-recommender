from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않\n'
 '습니다.- 1. 제14조(계약 전 알릴 의무)에 따라 계약자 또는 피보험자가 회사에 알린 내용\n'
 '- 이나 건강진단 내용이 보험금 지급사유의 발생에 영향을 미쳤음을 회사가 증명\n'
 '- 하는 경우\n'
 '- 2. 제16조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는\n'
 '- 경우\n'
 '- 3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우. 다만,\n'
 '- 진단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는'),
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
