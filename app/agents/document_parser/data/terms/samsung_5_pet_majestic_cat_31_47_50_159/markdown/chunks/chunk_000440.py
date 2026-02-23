from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑦ 제1항 내지 제4항의 창상봉합술은 「국민건강보험법」 에서 정한 요양급여 또는 「의\n'
 '- 료급여법」 에서 정한 의료급여 절차를 거쳐 급여항목이 발생한 경우를 말합니다.\n'
 '# 제4조 (보험금의 청구)① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서(진단서, 진료비세부내역서("건강보험심사평가원 진료수가코드(EDI)" 필\n'
 '- 수 기재), 진료기록부, 수술증명서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
