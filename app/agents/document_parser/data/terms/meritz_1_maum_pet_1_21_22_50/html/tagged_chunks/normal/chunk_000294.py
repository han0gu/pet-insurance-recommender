from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>계약자는 이 특별약관에 따라 보험료를 ( )회에 분할하여 회사에 납입합니다.</p><h1 "
 "id='21' style='font-size:14px'>제2조(나눠 내는 보험료의 납입)</h1><br><p id='22' "
 "data-category='paragraph' style='font-size:14px'>① 계약자는 계약을 체결할 때에 제1회 나눠 내는 "
 "보험료를 납입하고 제2회 이후의 나눠<br>내는 보험료는 아래에 기재된 납입기일까지 납입하여야 합니다.</p><br><p id='23'"),
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
 'indexing': {'chunk_id': 'chunk_000294',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
