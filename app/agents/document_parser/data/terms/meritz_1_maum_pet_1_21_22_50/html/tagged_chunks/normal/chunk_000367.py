from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 44 -</footer><p id='45' data-category='paragraph' "
 "style='font-size:14px'>환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계약 또는 보험료납<br>입영수증에 "
 "보험료 공제대상임이 표시된 보험의 보험료를 말한다.</p><p id='46' data-category='paragraph' "
 "style='font-size:14px'>2"),
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
 'indexing': {'chunk_id': 'chunk_000367',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
