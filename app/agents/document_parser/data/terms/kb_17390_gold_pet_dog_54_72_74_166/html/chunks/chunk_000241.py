from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약자가 회사로부터 보험계약대출을 받은 경우 계약이 해지되는 즉시 해약환</p><br><p id='52' "
 "data-category='list' style='font-size:16px'>급금에서 보험계약대출원금과 이자가 차감된다는 "
 '내용<br>\uf000 납입최고(독촉)기간의 마지막 날이 영업일이 아닌 때에는 최고(독촉)기간은 그 다<br>음 날까지로 '
 '합니다.<br>\uf000 보험수익자와 계약자가 다른 경우 보험수익자에게도 제1항에 따른 내용을 알려 드<br>립니다.<br>\uf000 '
 '회사가 제1항에 따른 납입최고(독촉) 등을 전자문서로 안내하고자 할'),
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
