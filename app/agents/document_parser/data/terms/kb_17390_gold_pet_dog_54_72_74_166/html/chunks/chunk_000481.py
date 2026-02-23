from langchain_core.documents import Document

chunk = Document(
    page_content=("상해입원일당으로 보험수익자에게 지급합니다.</p><br><p id='196' data-category='paragraph' "
 "style='font-size:16px'>\uf000 제1항의 상해입원일당의 지급일수는 1회 입원당 180일을 한도로 합니다.</p><p "
 "id='197' data-category='list' style='font-size:16px'>제2조(보험금 지급에 관한 "
 '세부규정)<br>\uf000 제1조(보험금의 지급사유)의 상해입원일당은 같은 상해의 치료를 목적으로 2회<br>이상 입원한 경우 이를 '
 '1회 입원으로 보아 각 입원일수를'),
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
