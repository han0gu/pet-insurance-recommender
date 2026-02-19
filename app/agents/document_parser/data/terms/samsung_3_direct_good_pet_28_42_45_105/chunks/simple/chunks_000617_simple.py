from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 회사는 계약자에게 갱신전 계약의 보험기간이 끝나기 15일 이전까지 갱신 요건, 보장 내용 변경내역, 갱신보험료 및 갱신 절차 등을 '
 '서면(등기우편 등), 전화(음성녹취) 또 는 전자문서(SMS 포함) 등으로 알려 드립니다.(단, 갱신보험료 납입면제가 발생한 경 우는 '
 '제외합니다.)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000617',
              'chunk_char_len': 159,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
