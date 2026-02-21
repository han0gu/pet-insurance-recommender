from langchain_core.documents import Document

chunk = Document(
    page_content=("보험<br>증권에 기재된 반려동물의 질병 또는 상해로 인한 손해를<br>보장하기 위하여 체결됩니다.</p><h1 id='75' "
 "style='font-size:18px'>제2조(용어의 정의)</h1><br><p id='76' "
 "data-category='paragraph' style='font-size:18px'>이 특별약관에서 사용되는 용어의 정의는, 이 "
 "특별약관의<br>다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.</p><br><h1 id='77' "
 "style='font-size:16px'>\uf000 계약관련"),
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
 'indexing': {'chunk_id': 'chunk_000267',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
