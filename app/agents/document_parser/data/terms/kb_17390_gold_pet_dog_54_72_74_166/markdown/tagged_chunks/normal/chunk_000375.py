from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 "창상봉합술(급여)"는 국민건강보험법에서 정한 요양급여 또는 의료급여\n'
 '법에서 정한 의료급여의 절차를 거쳐 급여항목이 발생한 경우에 한합니다.# 제4조(보험금의 청구)\uf000 보험수익자는 다음의 서류를 '
 '제출하고 보험금을 청구하여야 합니다.- 1. 청구서(회사 양식)\n'
 '- 2. 사고증명서(진료비세부내역서("건강보험심사평가원 진료수가코드(EDI)" 필수\n'
 '- 기재), 진단서, 진료기록부(검사기록지 포함) 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인 상'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000375',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
