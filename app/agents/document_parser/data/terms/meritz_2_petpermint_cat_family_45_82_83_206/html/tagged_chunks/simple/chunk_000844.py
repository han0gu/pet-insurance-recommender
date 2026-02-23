from langchain_core.documents import Document

chunk = Document(
    page_content=("사항은 보통약관 및 해당 특별<br>약관을 따릅니다.</p><footer id='9' "
 "style='font-size:14px'>168</footer><h1 id='10' style='font-size:14px'>【 별첨 "
 "】특정질병 분류표(반려묘)</h1><br><p id='11' data-category='paragraph' "
 "style='font-size:14px'>보험계약을 체결할 때 반려동물의 건강상태가 회사가 정한 기준에 적합<br>하지 않은 경우 또는 "
 '보험계약을 체결한 후 계약 전 알릴 의무 위반의<br>효과 등으로 보장을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000844',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
