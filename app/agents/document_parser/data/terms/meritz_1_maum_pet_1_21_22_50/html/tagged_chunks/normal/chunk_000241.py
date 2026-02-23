from langchain_core.documents import Document

chunk = Document(
    page_content=("회사의 제1항에 따른<br>지급보험금 결정에는 영향을 미치지 않습니다.</p><h1 id='69' "
 "style='font-size:14px'>제11조(손해방지의무)</h1><br><p id='70' "
 "data-category='paragraph' style='font-size:14px'>① 보험사고가 생긴 때에는 계약자 또는 피보험자는 "
 "아래의 사항을 이행하여야 합니다.</p><br><p id='71' data-category='list' "
 "style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000241',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
