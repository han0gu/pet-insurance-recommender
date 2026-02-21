from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 특<br>별약관과 다른 계약이 모두 의무보험인 경우에도 같습니다.</p><h1 id='60' "
 "style='font-size:14px'>손해액 ×</h1><br><h1 id='61' style='font-size:14px'>이 "
 "계약의 보상책임액</h1><br><p id='62' data-category='paragraph' "
 "style='font-size:14px'>다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액</p><h1 id='63' "
 "style='font-size:14px'>【사례】</h1><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000237',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
