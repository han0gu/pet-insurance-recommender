from langchain_core.documents import Document

chunk = Document(
    page_content=("하며 계약자에 따라 다르<br>게 해석하지 않습니다.</p><br><h1 id='109' "
 "style='font-size:14px'>【신의성실의 원칙】</h1><br><p id='110' "
 "data-category='paragraph' style='font-size:14px'>권리의 행사와 의무의 이행은 신의와 성실을 가지고 "
 "행동하여 상대방의 신뢰와 기대<br>를 배반하여서는 안 된다는 원칙(「민법」제2조 제1항)</p><br><p id='111' "
 "data-category='list' style='font-size:14px'>②"),
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
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
