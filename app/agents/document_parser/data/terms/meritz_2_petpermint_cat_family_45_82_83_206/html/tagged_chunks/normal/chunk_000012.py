from langchain_core.documents import Document

chunk = Document(
    page_content=(". 이 계약의 평 균공시이율은 2.75%입니다.</td></tr></tbody></table><footer id='16' "
 "style='font-size:14px'>48</footer><table id='17' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>계약자 "
 '적립액</td><td>장래의 해약환급금 등을 지급하기 위하여 계 약자가 납입한 보험료 중 일정액을 기준으로 보험료 및 해약환급금 '
 '산출방법서에서 정한 방법에 따라 계산한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
