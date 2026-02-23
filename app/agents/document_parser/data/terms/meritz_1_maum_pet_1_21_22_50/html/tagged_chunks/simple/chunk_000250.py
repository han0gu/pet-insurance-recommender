from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자의 법률상 손해배상책임을 확정하기 위하여 피보험자가 피해자와 행하<br>는 합의·절충·중재 또는 소송(확인의 소를 '
 "포함합니다)에 대하여 협조하거나, 피보험자<br>를 위하여 이러한 절차를 대행할 수 있습니다.</p><footer id='78' "
 "style='font-size:14px'>- 26 -</footer><p id='79' data-category='paragraph' "
 "style='font-size:14px'>② 회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
