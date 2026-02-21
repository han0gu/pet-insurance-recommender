from langchain_core.documents import Document

chunk = Document(
    page_content=('제 1항의 장애인증명서는 제출하지 않을 수 있습니다.<br>③ 장애인으로서 그 장애기간이 기재된 장애인증명서를 제1항 따라 회사에 제출한 '
 '때에<br>는 그 장애기간 동안은 이를 다시 제출하지 않을 수 있습니다.<br>④ 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 '
 "경우 계약자는 이를 회사에<br>알리고 변경된 장애기간이 기재된 장애인증명서를 제출하여야 합니다.</p><h1 id='66' "
 "style='font-size:14px'>제3조(장애인전용보험으로의 전환)</h1><br><p id='67'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000380',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
