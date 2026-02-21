from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는<br>계약자가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카<br>드의 매출을 취소하며 '
 "이자를 더하여 지급하지 않습니다.</p><h1 id='9' style='font-size:14px'>제20조(청약의 "
 "철회)</h1><br><p id='10' data-category='paragraph' style='font-size:14px'>① "
 '계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
