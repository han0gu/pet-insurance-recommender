from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자가 지급한 소송비용, 변호사비용, 중재, 화해 또는 조정에 관한 비용<br>라. 보험증권상 보상한도액내의 금액에 대한 '
 '공탁보증보험료. 그러나 회사는 그러한<br>보증을 제공할 책임은 부담하지 않습니다.<br>마'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000210',
              'chunk_char_len': 123,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
