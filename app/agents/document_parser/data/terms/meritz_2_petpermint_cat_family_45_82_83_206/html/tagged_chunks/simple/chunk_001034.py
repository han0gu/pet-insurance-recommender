from langchain_core.documents import Document

chunk = Document(
    page_content=('발목관절(족관절)을 말한다.<br>5) “한다리의 발목이상을 잃었을 때”라 함은 발목관절<br>(족관절)부터(발목관절 포함) 심장에 '
 '가까운 쪽에서<br>절단된 때를 말하며, 무릎관절(슬관절)의 상부에서<br>절단된 경우도 포함한다.<br>6) 다리의 관절기능 장해 '
 "평가는 다리의 3대관절의 관절</p><footer id='53' style='font-size:14px'>193</footer><p "
 "id='54' data-category='paragraph' style='font-size:20px'>운동범위 제한 및 "
 '무릎관절(슬관절)의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001034',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
