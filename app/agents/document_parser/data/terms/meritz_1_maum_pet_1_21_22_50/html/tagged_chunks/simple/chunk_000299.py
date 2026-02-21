from langchain_core.documents import Document

chunk = Document(
    page_content=("거래은행 지정계좌를 이용하여 보험료를 자동 납입합니다.</p><h1 id='31' "
 "style='font-size:14px'>제2조(자동납입 신청)</h1><br><p id='32' "
 "data-category='paragraph' style='font-size:14px'>계약자는 보험계약과 동시에 계약자의 거래은행 "
 "지정계좌를 이용하여 보험료를 자동 납입<br>하는 별첨 신청서를 작성합니다.</p><p id='33' "
 "data-category='paragraph' style='font-size:14px'>제3조(보험료의"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000299',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
