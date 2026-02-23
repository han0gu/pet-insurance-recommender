from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>운동범위 제한 및 무릎관절(슬관절)의 동요성 등으로<br>평가한다.</p><br><p "
 "id='55' data-category='list' style='font-size:16px'>가) 각 관절의 운동범위 측정은 "
 '장해평가시점의 ｢산<br>업재해보상보험법 시행규칙｣ 제47조 제1항 및 제<br>3항의 정상인의 신체 각 관절에 대한 평균 '
 '운동<br>가능영역을 기준으로 정상각도 및 측정방법 등을<br>따른다.<br>나) 관절기능장해가 신경손상으로 인한 경우에는 '
 '운<br>동범위 측정이 아닌 근력 및'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_001035',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
