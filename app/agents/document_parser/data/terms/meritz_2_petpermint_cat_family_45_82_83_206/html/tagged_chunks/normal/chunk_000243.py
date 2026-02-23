from langchain_core.documents import Document

chunk = Document(
    page_content=("원칙에 따라 공정하게 약관을 해석하<br>여야 하며 계약자에 따라 다르게 해석하지 않습니다.</p><br><h1 id='34' "
 "style='font-size:18px'>【 신의성실의 원칙 】</h1><br><p id='35' "
 "data-category='paragraph' style='font-size:16px'>권리의 행사와 의무의 이행은 신의와 성실을 가지고 "
 "행동<br>하여 상대방의 신뢰와 기대를 배반하여서는 안된다는 원칙<br>(「민법」제2조 제1항)</p><h1 id='36' "
 "style='font-size:18px'>【 민법"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
