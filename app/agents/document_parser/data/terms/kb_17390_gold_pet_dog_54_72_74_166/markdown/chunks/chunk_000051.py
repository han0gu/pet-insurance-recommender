from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제13조(대표자의 지정)\uf000 계약자 또는 보험수익자가2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니- 58 -다. 이 '
 '경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합\n'
 '니다.\n'
 '\uf000 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하\n'
 '여 회사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는\n'
 '보험수익자에게도 효력이 미칩니다.- \n'
 '| \uf000 계약자가 2명 | 이상인 경우에는 그 책임을 연대로 합니다. |\n'
 '| --- | --- |'),
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
