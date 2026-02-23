from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>【강제집행】</h1><br><p id='72' data-category='paragraph' "
 "style='font-size:14px'>사법상 또는 행정법상의 의무를 이행하지 아니하는 사람에 대하여 국가가 강제 권<br>력으로 그 "
 "의무의 이행하는 것을 말합니다.</p><br><h1 id='73' "
 "style='font-size:14px'>【담보권실행】</h1><br><p id='74' data-category='paragraph' "
 "style='font-size:14px'>담보권을 설정한"),
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
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
