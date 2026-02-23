from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 부활(효력회복)을 승낙한 때에 계약자는 부활<br>(효력회복)을 청약한 날까지의 연체된 보험료와 이에 대한<br>연체된 '
 "이자(보장보험료에 대해서 평균공시이율+1%로 계산<br>한 이자)를 더하여 납입하여야 합니다.</p><br><h1 id='67' "
 "style='font-size:20px'>【 부활(효력회복) 】</h1><br><p id='68' "
 "data-category='paragraph' style='font-size:16px'>보험료 납입을 연체하여 계약이 해지되고 계약자가 "
 '해약<br>환급금을 받지 않은 경우 회사가'),
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
 'indexing': {'chunk_id': 'chunk_000201',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
