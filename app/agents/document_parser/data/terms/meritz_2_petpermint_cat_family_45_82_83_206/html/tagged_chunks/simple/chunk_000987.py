from langchain_core.documents import Document

chunk = Document(
    page_content=('구분하여 각각<br>을 하나의 운동단위로 보며, 하나의 운동단위 내에<br>서 여러 개의 척추체(척추뼈 몸통)에 압박골절이 발<br>생한 '
 '경우에는 각 척추체(척추뼈 몸통)의 압박률을<br>합산하고, 두 개 이상의 운동단위에서 장해가 발생<br>한 경우에는 그 중 가장 높은 '
 "지급률을 적용한다.</p><br><p id='83' data-category='list' style='font-size:20px'>3) "
 '척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가<br>그 증상을 악화시킨 부분만큼, 즉 이 사고와의 관여<br>도를 산정하여'),
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
 'indexing': {'chunk_id': 'chunk_000987',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
