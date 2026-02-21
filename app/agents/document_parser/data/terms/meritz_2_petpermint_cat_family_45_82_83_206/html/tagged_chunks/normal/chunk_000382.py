from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 해지 전에<br>발생한 보험금 지급사유에 대하여 회사는 보상합니다.</p><br><p id='31' "
 "data-category='list' style='font-size:16px'>① 계약자(보험수익자와 계약자가 다른 경우 "
 '보험수익자<br>를 포함합니다)에게 납입최고(독촉)기간 내에 연체보<br>험료를 납입하여야 한다는 내용<br>② 납입최고(독촉)기간이 '
 '끝나는 날까지 보험료를 납입하<br>지 않을 경우 납입최고(독촉)기간이 끝나는 날의 다음<br>날에 계약이 해지된다는 내용(이 경우 '
 '계약이 해지되<br>는 때에는 즉시'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000382',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
