from langchain_core.documents import Document

chunk = Document(
    page_content=('제<br>한으로 물이나 이에 준하는 음료 이외는 섭취하지 못<br>하는 경우를 말한다.<br>3) “씹어먹는 기능에 뚜렷한 장해를 남긴 '
 "때”라 함은<br>아래의 경우 중 하나 이상에 해당되는 때를 말한다.</p><br><p id='41' "
 "data-category='list' style='font-size:20px'>가) 뚜렷한 개구(입을 벌림)운동 제한 또는 "
 '뚜렷한<br>저작(씹기)운동 제한으로 미음 또는 이에 준하는<br>정도의 음식물(죽 등)이외는 섭취하지 못하는 경<br>우<br>나) '
 '위‧아래턱(상ㆍ하악)의 가운데'),
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
 'indexing': {'chunk_id': 'chunk_000953',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
