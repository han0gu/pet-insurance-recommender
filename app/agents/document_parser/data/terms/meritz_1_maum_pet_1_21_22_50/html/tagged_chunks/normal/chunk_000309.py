from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제2종 단체</h1><br><p id='54' data-category='paragraph' "
 "style='font-size:14px'>비영리법인단체 또는 변호사회, 의사회 등 동업자단체로서 5인 이상의 구성원이 "
 "있는<br>단체</p><br><h1 id='55' style='font-size:14px'>3"),
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
 'indexing': {'chunk_id': 'chunk_000309',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
