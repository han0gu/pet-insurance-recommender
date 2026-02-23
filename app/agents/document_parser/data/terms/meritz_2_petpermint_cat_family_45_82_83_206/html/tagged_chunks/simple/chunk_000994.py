from langchain_core.documents import Document

chunk = Document(
    page_content=('휘어지는 증상) 또는<br>20° 이상의 척추측만증(척추가 옆으로 휘어지는<br>증상) 변형이 있을 때<br>나) 척추체(척추뼈 몸통) '
 '한 개의 압박률이 60%이상<br>인 경우 또는 한 운동단위 내에 두 개 이상 척추<br>체(척추뼈 몸통)의 압박골절로 각 '
 "척추체(척추뼈<br>몸통)의 압박률의 합이 90% 이상일 때</p><br><p id='5' data-category='paragraph' "
 "style='font-size:20px'>10) 뚜렷한 기형이란 다음 중 어느 하나에 해당하는 경<br>우를 말한다.</p><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000994',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
