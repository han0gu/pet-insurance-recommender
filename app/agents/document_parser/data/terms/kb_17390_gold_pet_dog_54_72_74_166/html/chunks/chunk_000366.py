from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이하 같습니다)에서 정한 3~100% 장해지급률에 해당하는 장해상태가<br>되었을 때에는 장해분류표에서 정한 장해지급률을 이 '
 '특별약관의 보험가입금액에 곱<br>하여 산출한 금액을 일반상해후유장해(3~100%) 보험금으로 보험수익자에게 '
 "지급합<br>니다.</p><h1 id='28' style='font-size:14px'>제2조(보험금 지급에 관한 "
 "세부규정)</h1><br><p id='29' data-category='paragraph' "
 "style='font-size:14px'>\uf000 제1조(보험금의 지급사유)에서 장해지급률이 상해"),
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
