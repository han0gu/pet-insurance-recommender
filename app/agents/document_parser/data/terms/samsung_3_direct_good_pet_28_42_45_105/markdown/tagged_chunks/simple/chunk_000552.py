from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서)\n'
 '- 5. 기타 지정대리청구인이 보험금의 수령에 필요하여 제출하는 서류\n'
 '# 제7조 (준용규정)- \n'
 '이 특별약관에서 정하지 않은 사항은 보험계약을 따릅니다.- 100 -- \n'
 '100 / 1814-4. 특정 신체부위·질병 보장제한부 인수 특별약관# 제 1조 (계약의 체결 및 효력)① 이 특별약관은 '
 '보험계약(특별약관이 부가된 경우에는 특별약관을 포함합니다. 이하\n'
 '「보험계약」이라 합니다)을 체결 또는 변경할 때 다음 각 호의 경우 보험계약자(이하'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000552',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
