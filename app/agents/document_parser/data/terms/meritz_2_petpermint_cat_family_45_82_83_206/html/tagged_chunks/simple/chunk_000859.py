from langchain_core.documents import Document

chunk = Document(
    page_content=('안과 '
 '질환</td></tr><tr><td>FGA005</td><td>초자체변성</td></tr><tr><td>FGA006</td><td>상공막염</td></tr><tr><td>FGA007</td><td>녹내장</td></tr><tr><td>FGA008</td><td>고양이 '
 '호산구성 각결막염</td></tr><tr><td>QBA001</td><td>눈곱 (원인 '
 '불명)</td></tr><tr><td>QBA002</td><td>결막 충혈 (원인 '
 '불명)</td></tr><tr><td>QBA003</td><td>눈 가려움증 (원인'),
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
            'risk_domains': ['digestive', 'eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000859',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
