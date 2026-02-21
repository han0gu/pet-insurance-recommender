from langchain_core.documents import Document

chunk = Document(
    page_content=(". 또한, 보장개시일을 계약일로 봅니다.</p><br><p id='44' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는<br>경우에는 보장을 하지 "
 "않습니다.</p><br><p id='45' data-category='list' style='font-size:16px'>① "
 '제15조(계약 전 알릴 의무)에 따라 계약자 또는 피보<br>험자가 회사에 알린 내용이나 건강진단 내용이 보험금<br>지급사유의 발생에 '
 '영향을 미쳤음을 회사가'),
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
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
