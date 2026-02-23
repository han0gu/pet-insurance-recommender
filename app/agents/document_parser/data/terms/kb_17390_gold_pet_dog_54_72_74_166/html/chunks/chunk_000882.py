from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자가 회사로부터 보험계약대출을 받은 경우 특별약관이 해지되는 즉시 해<br>약환급금에서 보험계약대출원금과 이자가 차감된다는 '
 "내용</p><br><p id='58' data-category='list' style='font-size:14px'>\uf000 "
 '납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그<br>다음 날까지로 합니다.<br>\uf000 보험수익자와 '
 '계약자가 다른 경우 보험수익자에게도 제1항에 따른 내용을 알려드<br>립니다.<br>\uf000 타인을 위한 계약의 경우 그 특정된 '
 '타인에게도 제1항에 따른 내용을'),
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
