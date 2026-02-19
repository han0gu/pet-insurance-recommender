from langchain_core.documents import Document

chunk = Document(
    page_content=('제13조(계약내용의 변경 등)\n'
 '\uf000 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서면 등으로 알리거나 보험증권 의 뒷면에 '
 '기재하여 드립니다.\n'
 '① 보험종목 ② 보험기간 ③ 보험료 납입주기, 납입방법 및 납입기간 ④ 계약자, 피보험자 ⑤ 보험가입금액, 보험료, 배상책임의 경우 '
 '보상한도액 등 기타 계약의 내용'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000237',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
