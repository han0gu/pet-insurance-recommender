from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 보험<br>금 지급을 위해 필요하다고 인정하는 경우 관련 서류를 요<br>청할 수 있습니다.</p><footer '
 "id='22' style='font-size:14px'>88</footer><p id='23' "
 "data-category='paragraph' style='font-size:20px'>【동물병원 보험금 자동청구】</p><br><p "
 "id='24' data-category='paragraph' style='font-size:16px'>지정된 동물병원에서 펫퍼민트 "
 'ID카드를 제시하고 진료<br>를 받은 경우,'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000296',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
