from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><br><p id='55' data-category='list' style='font-size:16px'>1) "
 '“외모”란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을<br>말한다.<br>2) “추상(추한 모습)장해”라 함은 '
 '성형수술(반흔(흉<br>터)성형술, 레이저치료 등 포함)을 시행한 후에도<br>영구히 남게 되는 상태의 추상(추한 모습)을 '
 '말한다.<br>3) “추상(추한 모습)을 남긴 때”라 함은 상처의 흔적,<br>화상 등으로 피부의 변색, 모발의 결손, 조직(뼈, '
 '피<br>부 등)의 결손 및 함몰'),
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
            'risk_domains': ['digestive', 'eye', 'head', 'skin']},
 'indexing': {'chunk_id': 'chunk_000967',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
