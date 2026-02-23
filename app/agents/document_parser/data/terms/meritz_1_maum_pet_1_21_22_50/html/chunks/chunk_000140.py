from langchain_core.documents import Document

chunk = Document(
    page_content=(". 또한, 보장개시일을 계약일로 봅니다.</p><br><h1 id='45' style='font-size:14px'>③ 회사는 제2항에도 "
 "불구하고 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않습니다.</h1><br><p id='46' "
 "data-category='list' style='font-size:14px'>1. 제15조(계약 전 알릴 의무)에 따라 계약자 또는 "
 '피보험자가 회사에 알린 내용이나<br>건강진단 내용이 보험금 지급사유의 발생에 영향을 미쳤음을 회사가 증명하는 경우<br>2'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
