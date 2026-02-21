from langchain_core.documents import Document

chunk = Document(
    page_content=('다른 사람의 계속적 인 도움이 없이는 식사를 전혀 할 수 없는 상 태(15%) - 숟가락 사용은 가능하나 젓가락 사용이 불가능 하여 '
 '음식물 섭취에 있어 부분적으로 다른 사 람의 도움이 필요한 상태(10%) - 독립적인 음식물 섭취는 가능하나 젓가락을 이 용하여 생선을 '
 "바르거나 음식물을 자르지는 못 하는 상태(5%)</td></tr></tbody></table><footer id='59' "
 "style='font-size:14px'>205</footer><table id='60'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001110',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
