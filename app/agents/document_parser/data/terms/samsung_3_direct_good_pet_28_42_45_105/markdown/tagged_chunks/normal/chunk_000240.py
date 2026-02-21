from langchain_core.documents import Document

chunk = Document(
    page_content=('- 지급사유를 발생시킨 경우\n'
 '- 2. 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과 다\n'
 '- 른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보\n'
 '험금 지급사유가 발생한 경우에는 보험금 지급에 영향을 미치지 않습니다.- 56 -56 / 181<용어풀이># [이미 발생한 보험금 '
 '지급사유에 대한 보험금 지급]계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류를 변조하여 보험금을 청구한 경우,'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000240',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
